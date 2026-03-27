import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleStreamFilter;

import java.util.regex.Pattern;

/**
 * Emailen testua normalizatu StringToWordVector aplikatu aurretik.
 */
public class TextNormalizer extends SimpleStreamFilter {

    private static final long serialVersionUID = 1L;

    // Patroiak: URL, Email, Letra suelteak, HTML eta puntuazioa
    private static final Pattern URL_PATTERN = Pattern.compile(
        "https?\\s*:\\s*/\\s*/\\s*\\S+|ftp\\s*:\\s*/\\s*/\\s*\\S+",
        Pattern.CASE_INSENSITIVE
    );

    private static final Pattern EMAIL_PATTERN = Pattern.compile(
        "[a-zA-Z0-9._%+\\-]+@[a-zA-Z0-9.\\-]+\\.[a-zA-Z]{2,}"
    );

    private static final Pattern SPACED_LETTERS_PATTERN = Pattern.compile(
        "\\b([a-z])([\\s.]+[a-z]){3,}\\b"
    );

    private static final Pattern HTML_ATTR_PATTERN = Pattern.compile(
        "\\b[a-z]+=[a-zA-Z0-9#]+\\b",
        Pattern.CASE_INSENSITIVE
    );

    private static final Pattern HTML_FRAGMENT_PATTERN = Pattern.compile(
        "\\b(tr|td|br|th|li|ul|ol|div|span|img|href|src|nbsp|bgcolor|cellpad){2,}\\b",
        Pattern.CASE_INSENSITIVE
    );

    private static final Pattern REPEATED_PUNCT_PATTERN = Pattern.compile(
        "([^a-zA-Z0-9\\s])\\1{2,}"
    );

    private static final Pattern MULTI_SPACE_PATTERN = Pattern.compile(
        "\\s{2,}"
    );

    @Override
    public String globalInfo() {
        return "Emailen testua normalizatzeko iragazkia.";
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.enableAllAttributes();
        result.enableAllClasses();
        result.enable(Capability.NO_CLASS);
        return result;
    }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat) {
        return new Instances(inputFormat, 0);
    }

    // Instantzia bakarra prozesatu eta testua normalizatu
    @Override
    protected Instance process(Instance instance) {
        Instance result = (Instance) instance.copy();

        for (int i = 0; i < instance.numAttributes(); i++) {
            if (instance.attribute(i).type() == Attribute.STRING
                    && !instance.isMissing(i)) {

                String text = instance.stringValue(i);
                text = normalize(text);

                result.setValue(i,
                    result.dataset().attribute(i).addStringValue(text));
            }
        }
        return result;
    }

    // NORMALIZAZIO LOGIKA:

    // Normalizazio urratsak hurrenkeran aplikatu
    public static String normalize(String text) {
        if (text == null || text.isEmpty()) return text;

        text = URL_PATTERN.matcher(text).replaceAll(" URLTOKEN ");
        text = EMAIL_PATTERN.matcher(text).replaceAll(" EMAILTOKEN ");
        text = collapseSpacedLetters(text);
        text = HTML_ATTR_PATTERN.matcher(text).replaceAll(" ");
        text = HTML_FRAGMENT_PATTERN.matcher(text).replaceAll(" ");
        text = REPEATED_PUNCT_PATTERN.matcher(text).replaceAll("$1");
        text = MULTI_SPACE_PATTERN.matcher(text).replaceAll(" ").trim();

        return text;
    }

    // Letra solteen sekuentziak lotu
    private static String collapseSpacedLetters(String text) {
        StringBuffer sb = new StringBuffer();
        java.util.regex.Matcher m = SPACED_LETTERS_PATTERN.matcher(text);
        while (m.find()) {
            String matched = m.group();
            String collapsed = matched.replaceAll("[\\s.]+", "");
            m.appendReplacement(sb, collapsed);
        }
        m.appendTail(sb);
        return sb.toString();
    }
}
